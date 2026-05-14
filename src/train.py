"""
Training pipeline for PathOmicDRP.

Phase 1: Omics-only baseline (3-modal: genomic + transcriptomic + proteomic)
Phase 2: Full model with histology (after H&E feature extraction)
"""

import os
import sys
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import roc_auc_score, r2_score, mean_squared_error, mean_absolute_error
from scipy.stats import pearsonr, spearmanr

from model import PathOmicDRP, get_default_config
from dataset import PathOmicDataset, collate_fn, load_data
from training_plots import save_loss_curves

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
PROJECT_ROOT = Path(os.environ.get("BRCA_DRUG_PRED_ROOT", Path(__file__).resolve().parents[1])).resolve()
DATA_ROOT = Path(os.environ.get("BRCA_DRUG_PRED_DATA_ROOT", PROJECT_ROOT / "data")).resolve()
LOSS_PLOT_DIR = PROJECT_ROOT / "results" / "loss_plot"


def _fmt_lr(lr: float) -> str:
    s = f"{lr:.0e}"
    base, exp = s.split('e')
    sign = exp[0]
    num = str(int(exp[1:]))
    return f"{base}e{sign}{num}"


# ---------------------------------------------------------------------------
# Training utilities
# ---------------------------------------------------------------------------

class EarlyStopping:
    def __init__(self, patience=15, min_delta=1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float('inf')
        self.early_stop = False

    def __call__(self, val_loss):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True


def train_one_epoch(model, dataloader, optimizer, criterion, device, lambda_kd: float = 0.0):
    model.train()
    total_loss = 0.0      # tracks regression loss only (comparable across configs)
    total_kd = 0.0
    n_samples = 0

    for batch in dataloader:
        genomic = batch['genomic'].to(device)
        transcriptomic = batch['transcriptomic'].to(device)
        proteomic = batch['proteomic'].to(device)
        target = batch['target'].to(device)

        histology = batch.get('histology')
        histo_mask = batch.get('histo_mask')
        if histology is not None:
            histology = histology.to(device)
            histo_mask = histo_mask.to(device)

        optimizer.zero_grad()
        output = model(genomic, transcriptomic, proteomic, histology, histo_mask)
        pred = output['prediction'].squeeze(-1)
        reg_loss = criterion(pred, target)

        loss = reg_loss
        kd_value = 0.0
        if lambda_kd > 0 and 'kd_loss' in output:
            kd_term = output['kd_loss']
            loss = loss + lambda_kd * kd_term
            kd_value = kd_term.item()

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], max_norm=1.0
        )
        optimizer.step()

        # EMA self-distillation: update teacher from updated student
        if hasattr(model, 'update_teacher'):
            model.update_teacher()

        bs = len(target)
        total_loss += reg_loss.item() * bs
        total_kd += kd_value * bs
        n_samples += bs

    avg_reg = total_loss / n_samples
    avg_kd = total_kd / n_samples
    if lambda_kd > 0:
        return {'reg_loss': avg_reg, 'kd_loss': avg_kd}
    return avg_reg


@torch.no_grad()
def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    n_samples = 0
    all_preds = []
    all_targets = []

    for batch in dataloader:
        genomic = batch['genomic'].to(device)
        transcriptomic = batch['transcriptomic'].to(device)
        proteomic = batch['proteomic'].to(device)
        target = batch['target'].to(device)

        histology = batch.get('histology')
        histo_mask = batch.get('histo_mask')
        if histology is not None:
            histology = histology.to(device)
            histo_mask = histo_mask.to(device)

        output = model(genomic, transcriptomic, proteomic, histology, histo_mask)
        pred = output['prediction'].squeeze(-1)
        loss = criterion(pred, target)

        total_loss += loss.item() * len(target)
        n_samples += len(target)
        all_preds.append(pred.cpu().numpy())
        all_targets.append(target.cpu().numpy())

    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)

    metrics = {
        'loss': total_loss / n_samples,
        'rmse': np.sqrt(mean_squared_error(all_targets, all_preds)),
        'mae': mean_absolute_error(all_targets, all_preds),
        'r2': r2_score(all_targets, all_preds),
    }

    if len(np.unique(all_targets)) > 1:
        pcc, _ = pearsonr(all_targets, all_preds)
        scc, _ = spearmanr(all_targets, all_preds)
        metrics['pcc'] = pcc
        metrics['scc'] = scc

    # Binary classification metrics (if applicable)
    if len(np.unique(all_targets)) == 2:
        metrics['auroc'] = roc_auc_score(all_targets, all_preds)

    return metrics, all_preds, all_targets


# ---------------------------------------------------------------------------
# Prepare clinical drug response as targets
# ---------------------------------------------------------------------------

def prepare_clinical_targets(base_dir=None):
    """Prepare binary drug response targets from TCGA clinical data.

    Responder: Complete Response + Partial Response
    Non-responder: Progressive Disease + Stable Disease
    """
    if base_dir is None:
        base_dir = DATA_ROOT
    drug_df = pd.read_csv(os.path.join(base_dir, "01_clinical/TCGA_BRCA_drug_treatments.csv"))

    # Filter for clear outcomes only
    response_map = {
        'Complete Response': 1,
        'Partial Response': 1,
        'Stable Disease': 0,
        'Progressive Disease': 0,
    }
    drug_df = drug_df[drug_df['treatment_outcome'].isin(response_map.keys())].copy()
    drug_df['response'] = drug_df['treatment_outcome'].map(response_map)

    # For patients with multiple treatment records, take the first
    patient_response = drug_df.groupby('submitter_id')['response'].first()

    print(f"Clinical targets: {len(patient_response)} patients")
    print(f"  Responder: {(patient_response == 1).sum()}")
    print(f"  Non-responder: {(patient_response == 0).sum()}")

    return patient_response


# ---------------------------------------------------------------------------
# Main training loop (5-fold CV)
# ---------------------------------------------------------------------------

def run_cross_validation(
    data: dict,
    targets: pd.Series,
    config: dict,
    n_folds: int = 5,
    n_epochs: int = 100,
    batch_size: int = 32,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    output_dir: str | os.PathLike = PROJECT_ROOT / "results",
    histology_dir: str = None,
):
    os.makedirs(output_dir, exist_ok=True)

    # Get patients with all 3 modalities + target
    gen_ids = set(data['genomic']['patient_id'])
    trans_ids = set(data['transcriptomic']['patient_id'])
    prot_ids = set(data['proteomic']['patient_id'])
    target_ids = set(targets.index)

    common = sorted(gen_ids & trans_ids & prot_ids & target_ids)
    print(f"Patients with all 3 modalities + target: {len(common)}")

    if len(common) < 20:
        print("Not enough patients for cross-validation. Trying 2-modal (genomic + transcriptomic)...")
        common = sorted(gen_ids & trans_ids & target_ids)
        print(f"Patients with 2 modalities + target: {len(common)}")

    targets_filtered = targets.loc[common]

    # Cross-validation
    if config.get('task') == 'classification' and len(np.unique(targets_filtered.values)) == 2:
        kf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
        split_iter = kf.split(common, targets_filtered.values)
    else:
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
        split_iter = kf.split(common)

    all_fold_metrics = []
    loss_histories = []

    for fold, (train_idx, val_idx) in enumerate(split_iter):
        print(f"\n{'='*60}")
        print(f"Fold {fold+1}/{n_folds}")
        print(f"{'='*60}")

        train_ids = [common[i] for i in train_idx]
        val_ids = [common[i] for i in val_idx]

        # Create datasets with scalers fit on training data
        train_ds = PathOmicDataset(
            train_ids, data['genomic'], data['transcriptomic'], data['proteomic'],
            targets=targets_filtered, histology_dir=histology_dir, fit_scalers=True,
        )
        val_ds = PathOmicDataset(
            val_ids, data['genomic'], data['transcriptomic'], data['proteomic'],
            targets=targets_filtered, histology_dir=histology_dir, scalers=train_ds.scalers,
        )

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                                   collate_fn=collate_fn, num_workers=2, drop_last=False)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                                 collate_fn=collate_fn, num_workers=2)

        # Model
        model = PathOmicDRP(config).to(DEVICE)
        # KD_GNN teacher params have requires_grad=False — exclude them so AdamW
        # doesn't allocate optimizer state for them.
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs, eta_min=lr * 0.01)
        lambda_kd = float(config.get('lambda_kd', 0.0)) if config.get('use_kd_gnn', False) else 0.0

        if config.get('task') == 'classification':
            # Weighted BCE for class imbalance
            pos_weight = torch.tensor([(targets_filtered == 0).sum() / max((targets_filtered == 1).sum(), 1)])
            criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(DEVICE))
        else:
            criterion = nn.HuberLoss(delta=1.0)

        early_stopping = EarlyStopping(patience=15)
        best_val_loss = float('inf')
        best_model_state = None
        loss_history = {'fold': fold + 1, 'epoch': [], 'train_loss': [], 'val_loss': []}

        for epoch in range(n_epochs):
            train_out = train_one_epoch(
                model, train_loader, optimizer, criterion, DEVICE, lambda_kd=lambda_kd
            )
            val_metrics, _, _ = evaluate(model, val_loader, criterion, DEVICE)
            scheduler.step()

            train_loss_value = train_out['reg_loss'] if isinstance(train_out, dict) else train_out
            loss_history['epoch'].append(epoch + 1)
            loss_history['train_loss'].append(float(train_loss_value))
            loss_history['val_loss'].append(float(val_metrics['loss']))

            if val_metrics['loss'] < best_val_loss:
                best_val_loss = val_metrics['loss']
                best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

            if (epoch + 1) % 10 == 0:
                metric_str = " | ".join(f"{k}: {v:.4f}" for k, v in val_metrics.items())
                if isinstance(train_out, dict):
                    train_str = (
                        f"Train reg: {train_out['reg_loss']:.4f} | "
                        f"KD: {train_out['kd_loss']:.4f}"
                    )
                else:
                    train_str = f"Train loss: {train_out:.4f}"
                print(f"  Epoch {epoch+1:3d} | {train_str} | Val {metric_str}")

            early_stopping(val_metrics['loss'])
            if early_stopping.early_stop:
                print(f"  Early stopping at epoch {epoch+1}")
                break

        # Load best model and evaluate
        model.load_state_dict(best_model_state)
        model.to(DEVICE)
        final_metrics, preds, targets_arr = evaluate(model, val_loader, criterion, DEVICE)

        print(f"\n  Fold {fold+1} Final: " + " | ".join(f"{k}: {v:.4f}" for k, v in final_metrics.items()))
        all_fold_metrics.append(final_metrics)

        # Save fold model
        torch.save(best_model_state, os.path.join(output_dir, f"fold{fold+1}_model.pt"))
        loss_histories.append(loss_history)

    # Aggregate results
    print(f"\n{'='*60}")
    print("CROSS-VALIDATION RESULTS")
    print(f"{'='*60}")
    avg_metrics = {}
    for key in all_fold_metrics[0]:
        values = [m[key] for m in all_fold_metrics]
        avg_metrics[key] = {'mean': np.mean(values), 'std': np.std(values)}
        print(f"  {key}: {np.mean(values):.4f} +/- {np.std(values):.4f}")

    # Save results
    with open(os.path.join(output_dir, "cv_results.json"), 'w') as f:
        json.dump({
            'config': config,
            'n_patients': len(common),
            'n_folds': n_folds,
            'fold_metrics': all_fold_metrics,
            'avg_metrics': avg_metrics,
        }, f, indent=2, default=str)
    plot_filename = f"phase1_omics_baseline_lr{_fmt_lr(lr)}.png"
    save_loss_curves(loss_histories, LOSS_PLOT_DIR, plot_filename, title="Phase 1 Loss Curves")

    return avg_metrics


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    _cfg_path = PROJECT_ROOT / "configs" / "train_phase1.json"
    with open(_cfg_path) as _f:
        _train_cfg = json.load(_f)

    print(f"Device: {DEVICE}")
    print(f"Loading data...")

    data = load_data()
    targets = prepare_clinical_targets()

    # Derive feature dims from the loaded matrices (excludes 'patient_id' column).
    genomic_dim = sum(1 for c in data['genomic'].columns if c != 'patient_id')
    n_pathways = sum(1 for c in data['transcriptomic'].columns if c != 'patient_id')
    proteomic_dim = sum(1 for c in data['proteomic'].columns if c != 'patient_id')
    print(f"Feature dims  | genomic: {genomic_dim} | transcriptomic: {n_pathways} | proteomic: {proteomic_dim}")

    # Phase 1: Omics-only baseline (classification: responder vs non-responder)
    config = get_default_config(
        genomic_dim=genomic_dim,
        n_pathways=n_pathways,
        proteomic_dim=proteomic_dim,
        n_drugs=_train_cfg.get('n_drugs', 1),
        use_histology=_train_cfg.get('use_histology', False),
    )
    config['task'] = _train_cfg.get('task', 'classification')
    config['modality_dropout'] = _train_cfg.get('modality_dropout', 0.15)

    print(f"\n{'='*60}")
    print("Phase 1: Omics-only Baseline (3-modal)")
    print(f"{'='*60}")

    results = run_cross_validation(
        data=data,
        targets=targets,
        config=config,
        n_folds=_train_cfg.get('n_folds', 5),
        n_epochs=_train_cfg.get('n_epochs', 100),
        batch_size=_train_cfg.get('batch_size', 32),
        lr=_train_cfg.get('lr', 5e-4),
        weight_decay=_train_cfg.get('weight_decay', 1e-4),
        output_dir=PROJECT_ROOT / "results" / "phase1_omics_baseline",
    )
