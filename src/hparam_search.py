"""
Optuna hyperparameter search for PathOmicDRP phases 1, 2, and 3.

Usage:
    python src/hparam_search.py 1                  # Phase 1, 30 trials (default)
    python src/hparam_search.py 2 --n-trials 20    # Phase 2, 20 trials
    python src/hparam_search.py 3

Each phase reads its base config from configs/train_phase{N}.json.
Best hyperparameters are merged into the base config and saved to
configs/train_phase{N}_best.json.
After the search, a full training run with the best params is executed
automatically.
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import optuna
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

optuna.logging.set_verbosity(optuna.logging.WARNING)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CONFIGS_DIR = PROJECT_ROOT / "configs"

# Lightweight proxy settings used during search (not final training)
HPO_EPOCHS   = 50
HPO_PATIENCE = 7


# ---------------------------------------------------------------------------
# Shared utilities
# ---------------------------------------------------------------------------

def _pearson_loss(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Differentiable mean per-drug Pearson loss: 1 - mean(corr)."""
    if pred.size(0) < 2:
        return pred.new_tensor(0.0)
    pc  = pred   - pred.mean(0, keepdim=True)
    tc  = target - target.mean(0, keepdim=True)
    num = (pc * tc).sum(0)
    den = (pc.pow(2).sum(0) * tc.pow(2).sum(0) + eps).sqrt()
    return 1.0 - (num / den).nan_to_num(0.0).mean()


def _quick_fit(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: nn.Module,
    lr: float,
    weight_decay: float,
    pearson_alpha: float = 0.0,
) -> float:
    """Short proxy training run; returns best validation loss."""
    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr, weight_decay=weight_decay,
    )
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=HPO_EPOCHS, eta_min=lr * 0.01
    )
    best_loss, no_improve = float("inf"), 0

    def _fwd(batch):
        g  = batch["genomic"].to(DEVICE)
        t  = batch["transcriptomic"].to(DEVICE)
        p  = batch["proteomic"].to(DEVICE)
        y  = batch["target"].to(DEVICE)
        h  = batch.get("histology")
        hm = batch.get("histo_mask")
        if h is not None:
            h, hm = h.to(DEVICE), hm.to(DEVICE)
        pred = model(g, t, p, h, hm)["prediction"]
        if pred.shape[-1] == 1 and y.dim() == 1:
            pred = pred.squeeze(-1)
        return pred, y

    for _ in range(HPO_EPOCHS):
        model.train()
        for batch in train_loader:
            opt.zero_grad()
            pred, y = _fwd(batch)
            loss = criterion(pred, y)
            if pearson_alpha > 0 and pred.dim() > 1:
                loss = loss + pearson_alpha * _pearson_loss(pred, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            if hasattr(model, "update_teacher"):
                model.update_teacher()

        model.eval()
        vl, n = 0.0, 0
        with torch.no_grad():
            for batch in val_loader:
                pred, y = _fwd(batch)
                vl += criterion(pred, y).item() * len(y)
                n  += len(y)
        vl /= max(n, 1)
        sch.step()

        if vl < best_loss - 1e-6:
            best_loss, no_improve = vl, 0
        else:
            no_improve += 1
        if no_improve >= HPO_PATIENCE:
            break

    return best_loss


def _suggest_common(trial: optuna.Trial) -> dict:
    return {
        "lr":               trial.suggest_float("lr",               1e-5, 1e-2, log=True),
        "weight_decay":     trial.suggest_float("weight_decay",     1e-6, 1e-2, log=True),
        "batch_size":       trial.suggest_categorical("batch_size", [16, 32, 64]),
        "modality_dropout": trial.suggest_float("modality_dropout", 0.0,  0.3),
        "hidden_dim":       trial.suggest_categorical("hidden_dim", [128, 256, 512]),
    }


# ---------------------------------------------------------------------------
# Phase 1 — 3-modal omics classification
# ---------------------------------------------------------------------------

def run_phase1_search(base_cfg: dict, n_trials: int) -> dict:
    from train import load_data, prepare_clinical_targets
    from model import PathOmicDRP, get_default_config
    from dataset import PathOmicDataset, collate_fn

    print("Loading Phase 1 data ...")
    data    = load_data()
    targets = prepare_clinical_targets()

    gen_ids   = set(data["genomic"]["patient_id"])
    trans_ids = set(data["transcriptomic"]["patient_id"])
    prot_ids  = set(data["proteomic"]["patient_id"])
    common    = sorted(gen_ids & trans_ids & prot_ids & set(targets.index))
    targets_s = targets.loc[common]

    g_dim = sum(1 for c in data["genomic"].columns       if c != "patient_id")
    t_dim = sum(1 for c in data["transcriptomic"].columns if c != "patient_id")
    p_dim = sum(1 for c in data["proteomic"].columns     if c != "patient_id")

    def objective(trial: optuna.Trial) -> float:
        hp = _suggest_common(trial)
        torch.manual_seed(trial.number)

        train_ids, val_ids = train_test_split(
            common, test_size=0.2, random_state=trial.number, stratify=targets_s.values
        )
        train_ds = PathOmicDataset(
            train_ids, data["genomic"], data["transcriptomic"], data["proteomic"],
            targets=targets_s, fit_scalers=True,
        )
        val_ds = PathOmicDataset(
            val_ids, data["genomic"], data["transcriptomic"], data["proteomic"],
            targets=targets_s, scalers=train_ds.scalers,
        )
        train_loader = DataLoader(
            train_ds, batch_size=hp["batch_size"], shuffle=True,
            collate_fn=collate_fn, num_workers=0,
        )
        val_loader = DataLoader(
            val_ds, batch_size=hp["batch_size"], shuffle=False,
            collate_fn=collate_fn, num_workers=0,
        )

        cfg = get_default_config(
            genomic_dim=g_dim, n_pathways=t_dim, proteomic_dim=p_dim,
            n_drugs=base_cfg["n_drugs"], use_histology=False,
        )
        cfg["task"]             = base_cfg["task"]
        cfg["modality_dropout"] = hp["modality_dropout"]
        cfg["hidden_dim"]       = hp["hidden_dim"]

        pos_w     = torch.tensor([(targets_s == 0).sum() / max((targets_s == 1).sum(), 1)])
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_w.to(DEVICE))
        model     = PathOmicDRP(cfg).to(DEVICE)
        return _quick_fit(model, train_loader, val_loader, criterion, hp["lr"], hp["weight_decay"])

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    return study.best_params


def run_final_phase1(best_params: dict, base_cfg: dict) -> None:
    from train import load_data, prepare_clinical_targets, run_cross_validation
    from model import get_default_config

    print("\nRunning final Phase 1 training with best params ...")
    data    = load_data()
    targets = prepare_clinical_targets()

    g_dim = sum(1 for c in data["genomic"].columns       if c != "patient_id")
    t_dim = sum(1 for c in data["transcriptomic"].columns if c != "patient_id")
    p_dim = sum(1 for c in data["proteomic"].columns     if c != "patient_id")

    cfg = get_default_config(
        genomic_dim=g_dim, n_pathways=t_dim, proteomic_dim=p_dim,
        n_drugs=base_cfg["n_drugs"], use_histology=False,
    )
    cfg["task"]             = base_cfg["task"]
    cfg["modality_dropout"] = best_params["modality_dropout"]
    cfg["hidden_dim"]       = best_params["hidden_dim"]

    run_cross_validation(
        data=data, targets=targets, config=cfg,
        n_folds=base_cfg["n_folds"],
        n_epochs=base_cfg["n_epochs"],
        batch_size=best_params["batch_size"],
        lr=best_params["lr"],
        weight_decay=best_params["weight_decay"],
        output_dir=PROJECT_ROOT / "results" / "phase1_hpo_best",
    )


# ---------------------------------------------------------------------------
# Phase 2 — multi-drug IC50 regression (3-modal)
# ---------------------------------------------------------------------------

def run_phase2_search(base_cfg: dict, n_trials: int) -> dict:
    import pandas as pd
    from train_phase2 import (
        ensure_imputed_ic50_targets, pick_available_drugs,
        MultiDrugDataset, require_file,
    )
    from model import PathOmicDRP, get_default_config

    BASE = PROJECT_ROOT / "data" / "07_integrated"
    print("Loading Phase 2 data ...")
    gen_df  = pd.read_csv(require_file(str(BASE / "X_genomic.csv"),       "genomic feature matrix"))
    tra_df  = pd.read_csv(require_file(str(BASE / "X_transcriptomic.csv"), "transcriptomic feature matrix"))
    pro_df  = pd.read_csv(require_file(str(BASE / "X_proteomic.csv"),      "proteomic feature matrix"))
    ensure_imputed_ic50_targets(gen_df)
    ic50_df = pd.read_csv(
        require_file(str(BASE / "predicted_IC50_all_drugs.csv"), "IC50 target matrix"),
        index_col=0,
    )

    n_drugs   = base_cfg.get("n_drugs", 13)
    drug_cols = pick_available_drugs(ic50_df, n_drugs)
    common    = sorted(
        set(gen_df["patient_id"]) & set(tra_df["patient_id"])
        & set(pro_df["patient_id"]) & set(ic50_df.index)
    )

    g_dim = len([c for c in gen_df.columns if c != "patient_id"])
    t_dim = len([c for c in tra_df.columns if c != "patient_id"])
    p_dim = len([c for c in pro_df.columns if c != "patient_id"])

    def objective(trial: optuna.Trial) -> float:
        hp = _suggest_common(trial)
        hp["pearson_alpha"] = trial.suggest_float("pearson_alpha", 0.0, 0.5)
        torch.manual_seed(trial.number)

        train_ids, val_ids = train_test_split(common, test_size=0.2, random_state=trial.number)
        train_ds = MultiDrugDataset(train_ids, gen_df, tra_df, pro_df, ic50_df, drug_cols, fit=True)
        val_ds   = MultiDrugDataset(val_ids,   gen_df, tra_df, pro_df, ic50_df, drug_cols,
                                     scalers=train_ds.scalers)
        train_loader = DataLoader(train_ds, batch_size=hp["batch_size"], shuffle=True,  num_workers=0,
                                   drop_last=len(train_ids) > hp["batch_size"])
        val_loader   = DataLoader(val_ds,   batch_size=hp["batch_size"], shuffle=False, num_workers=0)

        cfg = get_default_config(
            genomic_dim=g_dim, n_pathways=t_dim, proteomic_dim=p_dim,
            n_drugs=len(drug_cols), use_histology=False,
        )
        cfg["task"]             = "regression"
        cfg["modality_dropout"] = hp["modality_dropout"]
        cfg["hidden_dim"]       = hp["hidden_dim"]

        criterion = nn.HuberLoss(delta=1.0)
        model     = PathOmicDRP(cfg).to(DEVICE)
        return _quick_fit(
            model, train_loader, val_loader, criterion,
            hp["lr"], hp["weight_decay"], pearson_alpha=hp["pearson_alpha"],
        )

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    return study.best_params


def run_final_phase2(best_params: dict, base_cfg: dict) -> None:
    from train_phase2 import run_experiment

    print("\nRunning final Phase 2 training with best params ...")
    run_experiment(
        modalities=("genomic", "transcriptomic", "proteomic"),
        n_drugs=base_cfg["n_drugs"],
        n_folds=base_cfg["n_folds"],
        n_epochs=base_cfg["n_epochs"],
        batch_size=best_params["batch_size"],
        lr=best_params["lr"],
        weight_decay=best_params["weight_decay"],
        pearson_alpha=best_params.get("pearson_alpha", base_cfg.get("pearson_alpha", 0.2)),
        patience=base_cfg.get("patience", 20),
        modality_dropout=best_params["modality_dropout"],
        hidden_dim=best_params["hidden_dim"],
        tag="hpo_best_3modal_full",
    )


# ---------------------------------------------------------------------------
# Phase 3 — 4-modal (HPO uses 3-modal proxy for speed)
# ---------------------------------------------------------------------------

def run_phase3_search(base_cfg: dict, n_trials: int) -> dict:
    import pandas as pd
    from train_phase3_4modal import MultiDrugDataset4Modal
    from train_phase2 import ensure_imputed_ic50_targets, pick_available_drugs, require_file
    from model import PathOmicDRP, get_default_config

    BASE = PROJECT_ROOT / "data" / "07_integrated"
    print("Loading Phase 3 data (3-modal proxy — histology excluded from search) ...")
    gen_df  = pd.read_csv(require_file(str(BASE / "X_genomic.csv"),       "genomic feature matrix"))
    tra_df  = pd.read_csv(require_file(str(BASE / "X_transcriptomic.csv"), "transcriptomic feature matrix"))
    pro_df  = pd.read_csv(require_file(str(BASE / "X_proteomic.csv"),      "proteomic feature matrix"))
    ensure_imputed_ic50_targets(gen_df)
    ic50_df = pd.read_csv(
        require_file(str(BASE / "predicted_IC50_all_drugs.csv"), "IC50 target matrix"),
        index_col=0,
    )

    n_drugs   = base_cfg.get("n_drugs", 13)
    drug_cols = pick_available_drugs(ic50_df, n_drugs)
    common    = sorted(
        set(gen_df["patient_id"]) & set(tra_df["patient_id"])
        & set(pro_df["patient_id"]) & set(ic50_df.index)
    )

    g_dim = len([c for c in gen_df.columns if c != "patient_id"])
    t_dim = len([c for c in tra_df.columns if c != "patient_id"])
    p_dim = len([c for c in pro_df.columns if c != "patient_id"])

    def objective(trial: optuna.Trial) -> float:
        hp = _suggest_common(trial)
        hp["pearson_alpha"] = trial.suggest_float("pearson_alpha", 0.0, 0.5)
        torch.manual_seed(trial.number)

        train_ids, val_ids = train_test_split(common, test_size=0.2, random_state=trial.number)
        train_ds = MultiDrugDataset4Modal(
            train_ids, gen_df, tra_df, pro_df, ic50_df, drug_cols,
            histo_dir=None, fit=True,
        )
        val_ds = MultiDrugDataset4Modal(
            val_ids, gen_df, tra_df, pro_df, ic50_df, drug_cols,
            histo_dir=None, scalers=train_ds.scalers,
        )
        train_loader = DataLoader(train_ds, batch_size=hp["batch_size"], shuffle=True,  num_workers=0,
                                   drop_last=len(train_ids) > hp["batch_size"])
        val_loader   = DataLoader(val_ds,   batch_size=hp["batch_size"], shuffle=False, num_workers=0)

        cfg = get_default_config(
            genomic_dim=g_dim, n_pathways=t_dim, proteomic_dim=p_dim,
            n_drugs=len(drug_cols), use_histology=False,
        )
        cfg["task"]             = "regression"
        cfg["modality_dropout"] = hp["modality_dropout"]
        cfg["hidden_dim"]       = hp["hidden_dim"]

        criterion = nn.HuberLoss(delta=1.0)
        model     = PathOmicDRP(cfg).to(DEVICE)
        return _quick_fit(
            model, train_loader, val_loader, criterion,
            hp["lr"], hp["weight_decay"], pearson_alpha=hp["pearson_alpha"],
        )

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    return study.best_params


def run_final_phase3(best_params: dict, base_cfg: dict) -> None:
    from train_phase3_4modal import run_experiment

    print("\nRunning final Phase 3 training with best params ...")
    bs   = best_params["batch_size"]
    shared = dict(
        n_drugs=base_cfg["n_drugs"],
        n_folds=base_cfg["n_folds"],
        n_epochs=base_cfg["n_epochs"],
        lr=best_params["lr"],
        weight_decay=best_params["weight_decay"],
        pearson_alpha=best_params.get("pearson_alpha", base_cfg.get("pearson_alpha", 0.2)),
        patience=base_cfg.get("patience", 20),
        modality_dropout=best_params["modality_dropout"],
        hidden_dim=best_params["hidden_dim"],
    )
    # 3-modal baseline with best params
    run_experiment(use_histology=False, batch_size=bs, tag="hpo_best_3modal", **shared)
    # 4-modal with best params
    run_experiment(use_histology=True,  batch_size=bs, tag="hpo_best_4modal", **shared)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Optuna HPO for PathOmicDRP — saves best config and runs final training."
    )
    parser.add_argument("phase", type=int, choices=[1, 2, 3],
                        help="Training phase to optimise (1, 2, or 3)")
    parser.add_argument("--n-trials", type=int, default=30,
                        help="Number of Optuna trials (default: 30)")
    args = parser.parse_args()

    cfg_names = {
        1: "train_phase1.json",
        2: "train_phase2.json",
        3: "train_phase3_4modal.json",
    }
    cfg_name = cfg_names[args.phase]
    with open(CONFIGS_DIR / cfg_name) as f:
        base_cfg = json.load(f)

    print(f"=== Phase {args.phase} HPO | {args.n_trials} trials | device: {DEVICE} ===\n")

    search_fn = {1: run_phase1_search, 2: run_phase2_search, 3: run_phase3_search}
    best_params = search_fn[args.phase](base_cfg, args.n_trials)

    # Persist: merge best params into base config
    best_cfg = dict(base_cfg)
    best_cfg.update(best_params)
    best_cfg["_hpo_n_trials"] = args.n_trials
    # For phase 3 keep both batch_size keys consistent
    if args.phase == 3 and "batch_size" in best_params:
        best_cfg["batch_size_3modal"] = best_params["batch_size"]
        best_cfg["batch_size_4modal"] = best_params["batch_size"]

    best_cfg_path = CONFIGS_DIR / cfg_name.replace(".json", "_best.json")
    with open(best_cfg_path, "w") as f:
        json.dump(best_cfg, f, indent=2)

    print(f"\nBest params ({args.n_trials} trials):")
    print(json.dumps(best_params, indent=2))
    print(f"Saved to: {best_cfg_path}\n")

    # Final full training
    final_fn = {1: run_final_phase1, 2: run_final_phase2, 3: run_final_phase3}
    final_fn[args.phase](best_params, base_cfg)

    print("\nDone.")


if __name__ == "__main__":
    main()
